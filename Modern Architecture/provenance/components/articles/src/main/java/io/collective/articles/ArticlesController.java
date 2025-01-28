package io.collective.articles;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.collective.restsupport.BasicHandler;
import org.eclipse.jetty.server.Request;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import java.util.List;
import java.util.stream.Collectors;

public class ArticlesController extends BasicHandler {
    private final ArticleDataGateway gateway;
    private final ObjectMapper mapper;
    public ArticlesController(ObjectMapper mapper, ArticleDataGateway gateway) {
        super(mapper);
        this.gateway = gateway;
        this.mapper = mapper;
    }

    @Override
    public void handle(String target, Request request, HttpServletRequest servletRequest, HttpServletResponse servletResponse) {
        get("/articles", List.of("application/json", "text/html"), request, servletResponse, () -> {

            { // todo - query the articles gateway for *all* articles, map record to infos, and send back a collection of article infos
                List<ArticleInfo> infos =gateway.findAll().stream()
                        .map(r->new ArticleInfo(r.getId(), r.getTitle()))
                        .collect(Collectors.toList())
                        ;
                // send out response
                try {
                    servletResponse.setContentType("application/json");
                    servletResponse.setStatus(200);
                    mapper.writeValue(servletResponse.getWriter(),infos);
                } catch (Exception e) {
                    servletResponse.setStatus(500);
                    e.printStackTrace();
                }
            }
        });

        get("/available", List.of("application/json"), request, servletResponse, () -> {

            { // todo - query the articles gateway for *available* articles, map records to infos, and send back a collection of article infos
                List<ArticleInfo> infos =gateway.findAll().stream()
                        .filter(ArticleRecord::isAvailable)
                        .map(r->new ArticleInfo(r.getId(), r.getTitle()))
                        .collect(Collectors.toList())
                        ;
                // send out response
                try {
                    servletResponse.setContentType("application/json");
                    servletResponse.setStatus(200);
                    mapper.writeValue(servletResponse.getWriter(),infos);
                } catch (Exception e) {
                    servletResponse.setStatus(500);
                    e.printStackTrace();
                }
            }
        });
    }
}
